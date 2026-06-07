## MODIFIED Requirements

### Requirement: Plan action sequence from current frame to goal frame
The system SHALL plan an action sequence that transitions from the current state to the goal state using CEM optimization over latent-space rollouts, with optional warm-starting from an auxiliary policy network and optional gradient-based refinement.

#### Scenario: Basic goal-conditioned planning
- **WHEN** a current frame and goal frame are provided
- **THEN** the system encodes both frames, runs CEM optimization to find actions minimizing final latent distance to the goal, and returns the best action sequence

#### Scenario: Planning with warm-start initialization
- **WHEN** an auxiliary policy network is provided and a current frame and goal frame are given
- **THEN** the system uses the policy network's output as the initial mean for CEM optimization instead of zero-mean initialization

#### Scenario: Planning with gradient refinement
- **WHEN** gradient refinement is enabled
- **THEN** the system refines the CEM-optimized action sequence using gradient descent on the cost function before returning the result

#### Scenario: Planning with subgoal target
- **WHEN** a subgoal latent is provided instead of a goal frame
- **THEN** the system uses the subgoal latent directly as the planning target, skipping goal frame encoding

#### Scenario: Replanning trigger
- **WHEN** the planner returns a plan with a `needs_replan` flag set to True
- **THEN** the caller SHALL request a new plan, potentially with a new subgoal from the high-level planner

#### Scenario: Return uncertainty estimate
- **WHEN** the planner is configured with an ensemble world model
- **THEN** the system returns an uncertainty estimate alongside the action sequence and cost
