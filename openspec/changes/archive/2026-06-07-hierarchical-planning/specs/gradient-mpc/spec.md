## ADDED Requirements

### Requirement: Differentiable latent rollout
The system SHALL support differentiable latent rollouts through the world model by maintaining gradient flow across the autoregressive prediction chain, enabling gradient computation with respect to action sequences.

#### Scenario: Gradient flow through rollout
- **WHEN** a differentiable rollout is requested with `gradient_policy="straight_through"`
- **THEN** the system maintains gradient flow through all prediction steps, allowing backpropagation from final latent to action sequence

#### Scenario: Gradient clipping during rollout
- **WHEN** gradients through the rollout exceed a configurable threshold
- **THEN** the system clips gradients to prevent divergence

### Requirement: Gradient-based action refinement
The system SHALL refine an initial action sequence using gradient descent to minimize the latent-space distance between the rollout endpoint and the goal latent.

#### Scenario: Refine CEM output with gradient descent
- **WHEN** an initial action sequence is provided (e.g., from CEM)
- **THEN** the system performs gradient descent on the action sequence to further reduce the cost function

#### Scenario: Configurable refinement steps
- **WHEN** a number of gradient refinement steps is specified (default 10)
- **THEN** the system performs exactly that many gradient descent iterations with configurable learning rate

#### Scenario: Action bounds enforcement
- **WHEN** gradient updates push actions outside valid bounds
- **THEN** the system clamps actions to the configured [action_low, action_high] range after each step

### Requirement: Warm-starting from auxiliary network
The system SHALL support warm-starting the optimization with an initial action mean from an auxiliary value/policy network (Dreamer-style), when available.

#### Scenario: Warm-start with policy network output
- **WHEN** an auxiliary policy network is available and provides an action distribution for the current state
- **THEN** the system uses the policy network's mean action as the initial mean for CEM optimization

#### Scenario: Fallback without auxiliary network
- **WHEN** no auxiliary network is available
- **THEN** the system falls back to zero-mean initialization (existing CEM behavior)

### Requirement: Gradient MPC configuration
The system SHALL support configurable gradient MPC parameters including learning rate, number of refinement steps, gradient clip norm, and whether to enable warm-starting.

#### Scenario: Custom gradient MPC configuration
- **WHEN** a GradientMPCConfig is provided with custom parameters
- **THEN** the system uses those parameters for the gradient refinement stage

#### Scenario: Configuration validation
- **WHEN** invalid parameters are provided (e.g., negative learning rate)
- **THEN** the system raises a validation error
